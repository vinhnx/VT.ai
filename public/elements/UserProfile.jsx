import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"

export default function UserProfile() {
	return (
		<Card className="w-full max-w-md mx-auto mt-6">
			<CardHeader className="flex flex-col items-center gap-2">
                <Avatar>
                  <AvatarImage src={props.avatar_url || '/public/avatars/assistant.jpg'} alt="Avatar"/>
                  <AvatarFallback>VT</AvatarFallback>
                </Avatar>

				<CardTitle className="text-xl">{props.full_name || 'Anonymous'}</CardTitle>
				<div className="text-sm text-muted-foreground">{props.email}</div>
				<Badge variant="outline">{props.subscription_tier || 'free'}</Badge>
			</CardHeader>
			<CardContent>
				<div className="flex flex-col gap-2">
					<div><b>User ID:</b> {props.user_id}</div>
					<div><b>Provider:</b> {props.provider}</div>
					<div><b>Tokens Used:</b> {props.tokens_used}</div>
					<div><b>Joined:</b> {props.created_at?.slice(0, 10)}</div>
					<div className="flex items-center gap-2">
						<div className="font-medium">
							<b>Credits:</b> {props.credits_left ?? '-'} / {props.max_credits ?? '-'}
						</div>
					</div>
					{typeof props.credits_left === 'number' && typeof props.max_credits === 'number' && (
						<div className="w-full">
							<Progress
								className="w-full h-2 mt-1 bg-muted"
								value={Math.round((props.credits_left / props.max_credits) * 100)}
								indicatorClassName={
									props.credits_left === 0
										? 'bg-destructive'
									: props.credits_left < props.max_credits / 3
										? 'bg-yellow-500'
										: 'bg-green-500'
								}
							/>
							<div className="text-xs text-muted-foreground mt-1 text-right">
								{props.credits_left} / {props.max_credits}
							</div>
						</div>
					)}
					{props.reset_time && (
						<div className="text-xs text-muted-foreground mt-1">
							Next reset: {new Date(props.reset_time).toLocaleString('en-US', {
								month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false
							})} UTC
						</div>
					)}
				</div>
			</CardContent>
		</Card>
	)
}
